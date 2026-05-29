# Variant of 88bcf3b4
# Dimension: D2 — 边界遍历方向 (path)
# Change: 沿障碍物边界的遍历方向由 bar 末端到障碍物最近点的位置关系隐式决定
#         （根据 root side 选择 FL 或 FR 入场）改为左上角图例显式指定，所有障碍物统一按图例方向绕行。
#
# 设计分析
# 1. 具体化与行为追踪
#    - 原题输入里每个 triple 只有 bar / irregular segment / obstacle。输出中 segment 变成单调的
#      `FL*F*FR*` 或 `FR*F*FL*` 包裹路径，其中“先走哪一侧”来自输入 segment root 在 terminal 左/右。
#    - 本变体在左上角固定加入一个 2x2 单色 L 图例：
#        `[[X, X], [X, bg]]` -> CCW -> 统一使用 `FL*F*FR*`
#        `[[X, X], [bg, X]]` -> CW  -> 统一使用 `FR*F*FL*`
#      图例保留到输出中不变。
#    - 小例子：若某条 bar 从上边缘向下伸，输入 root 在 terminal 右侧。
#      原题会优先走 `FR`；现在如果左上角图例是 `CCW`，则即使 root 在右侧，输出也必须走
#      `FL*F*FR*`。因此“root side != 绕行方向”成为合法且常见的情况，D2 差异可见。
#
# 2. 复杂度校准
#    - 图例只提供两个档位 `CW/CCW`，不会引入过多自由度。
#    - 每张图里所有 triple 共享同一个图例，训练样例中只需比较 2-4 组图就能发现：
#      同图内所有 rope 的首段绕行方向一致，而不是各自跟 root side 绑定。
#    - 为避免方向变化太弱，generator 约束 `p1>=1` 且 `p3>=1`，让首段和末段对角都出现。
#
# 3. 隐式假设审查（相对原 main）
#    - 原代码 `off_r/off_c = seg_start - terminal`，后续 `pick_p_near + shifted` 默认保持同侧入场；
#      这是旧 D2 的核心前提，变体后不再成立，所以整段 skin-BFS 路由逻辑需要替换。
#    - 原代码最终只断言 pattern 是 `FL*F*FR*` 或 `FR*F*FL*` 之一，但不区分该选哪一个；
#      变体后必须由图例唯一决定，因此 solver 要在候选路径中按图例方向搜索并打分。
#    - 原代码可把所有非 bg 颜色都当对象色。变体增加角落图例后，solver 必须先解析并排除图例颜色，
#      否则会把图例误识别成 obstacle / segment。
#
# 4. 放置合法性
#    - generator 在左上角保留 3x3 空区，只放图例，不允许 triple bbox 进入。
#    - obstacle 不是任意 grow，而是先强制包含一段“贴着输出路径内侧”的支撑像素；
#      这样 solver 搜索时可用“贴边支撑长度”稳定选回正确候选路径。
#    - 仍保留原题 nearest-obstacle 约束：对每个 terminal，它自己的 obstacle 比其他 obstacle 更近。
#
# 5. 视觉可观察性
#    - 图例是显式可见的角落符号，不依赖颜色值大小。
#    - 同一网格中多个 triple 共享同一方向；不同样例中图例可能翻转，输出首段绕行也随之翻转。
#    - 输入 root side 继续随机，因此观察者能看出“不是 root side 在控制方向，而是角落符号在控制”。
#
# 6. 可视化整洁度
#    - 图例固定在左上角 2x2，小且远离主体，不会压住对象。
#    - obstacle 仍是小型 4-connected sprite，但至少沿输出路径内侧形成一小段连续贴边结构，
#      让 rope hugging 行为更清楚。
#
# 7. 颜色变化可推断性
#    - 仅新增 1 个独立图例色；它不参与排序或映射，只用于位置+形状编码 `CW/CCW`。
#    - bar / segment / obstacle / legend 四类颜色在单张图中两两不同，避免歧义。

import random
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


def _segment_main_path(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    sset = set(seg_pts)

    def neighbors(p: tuple[int, int]):
        r, c = p
        for q in neighbors4(r, c):
            if q in sset:
                yield q
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            q = (r + dr, c + dc)
            if q not in sset:
                continue
            mid1 = (r, c + dc)
            mid2 = (r + dr, c)
            if mid1 in sset or mid2 in sset:
                continue
            yield q

    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dist: dict[tuple[int, int], int] = {start: 0}
    while q:
        cur = q.popleft()
        for nxt in neighbors(cur):
            if nxt in prev:
                continue
            prev[nxt] = cur
            dist[nxt] = dist[cur] + 1
            q.append(nxt)

    far = max(dist.keys(), key=lambda p: (dist[p], p[0], p[1]))
    path = []
    cur = far
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _segment_degrees(seg_pts: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    sset = set(seg_pts)
    deg = {p: 0 for p in sset}
    for r, c in sset:
        for q in neighbors4(r, c):
            if q in sset:
                deg[(r, c)] += 1
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            q = (r + dr, c + dc)
            if q not in sset:
                continue
            mid1 = (r, c + dc)
            mid2 = (r + dr, c)
            if mid1 in sset or mid2 in sset:
                continue
            deg[(r, c)] += 1
    return deg


def _parse_corner_legend(g: np.ndarray, bg: int) -> tuple[str, int, set[tuple[int, int]]]:
    assert g.shape[0] >= 2 and g.shape[1] >= 2, "grid too small for legend"
    block = g[:2, :2]
    colors = sorted(set(int(v) for v in np.unique(block)) - {bg})
    assert len(colors) == 1, "legend must use one dedicated color"
    legend_color = colors[0]
    cells = {(r, c) for r in range(2) for c in range(2) if int(block[r, c]) == legend_color}
    assert len(cells) == 3 and (0, 0) in cells and (0, 1) in cells, "invalid legend shape"
    if (1, 0) in cells and (1, 1) not in cells:
        return "CCW", legend_color, cells
    if (1, 1) in cells and (1, 0) not in cells:
        return "CW", legend_color, cells
    raise AssertionError("legend must be top-left L or mirrored-L")


def _bar_fwd(comp: list[tuple[int, int]], h: int, w: int) -> tuple[int, int]:
    e0, e1 = _bar_endpoints(comp)
    border_ep = e0 if _touches_border(e0, h, w) else e1
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        return (1, 0) if border_ep[0] == 0 else (-1, 0)
    return (0, 1) if border_ep[1] == 0 else (0, -1)


def _step_classes(route: list[tuple[int, int]], fwd: tuple[int, int]) -> list[str]:
    classes = []
    for j in range(len(route) - 1):
        dr = route[j + 1][0] - route[j][0]
        dc = route[j + 1][1] - route[j][1]
        fwd_comp = dr * fwd[0] + dc * fwd[1]
        perp_comp = dr * fwd[1] + dc * (-fwd[0])
        if abs(dr) + abs(dc) == 1:
            classes.append("F" if fwd_comp > 0 else "X")
        elif fwd_comp > 0 and perp_comp > 0:
            classes.append("FR")
        elif fwd_comp > 0 and perp_comp < 0:
            classes.append("FL")
        else:
            classes.append("X")
    return classes


def _build_candidate_route(
    start: tuple[int, int],
    fwd: tuple[int, int],
    turn_dir: str,
    L: int,
    p1: int,
    p2: int,
    p3: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    right = (fwd[1], -fwd[0])
    left = (-right[0], -right[1])
    first_lat = right if turn_dir == "CW" else left
    second_lat = left if turn_dir == "CW" else right
    inner = (-first_lat[0], -first_lat[1])

    steps = [((fwd[0] + first_lat[0]), (fwd[1] + first_lat[1]))] * p1
    steps += [fwd] * p2
    steps += [((fwd[0] + second_lat[0]), (fwd[1] + second_lat[1]))] * p3
    assert len(steps) == L - 1

    route = [start]
    cur = start
    for dr, dc in steps:
        cur = (cur[0] + dr, cur[1] + dc)
        route.append(cur)

    support = [(route[i][0] + inner[0], route[i][1] + inner[1]) for i in range(p1, p1 + p2 + 1)]
    return route, support


def _route_score(
    g: np.ndarray,
    route: list[tuple[int, int]],
    support: list[tuple[int, int]],
    seg_col: int,
    obstacle_set: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[int, int, int, int, int] | None:
    h, w = g.shape
    for p in route:
        r, c = p
        if not (0 <= r < h and 0 <= c < w):
            return None
        if p in obstacle_set or p in reserved:
            return None
        if int(g[r, c]) not in (get_bg(g), seg_col):
            return None

    support_hits = 0
    support_run = 0
    for p in support:
        if p in obstacle_set:
            support_hits += 1
            support_run += 1
        else:
            break
    if support_run == 0:
        return None

    extra_touches = 0
    for r, c in route:
        if any((nr, nc) in obstacle_set for nr, nc in neighbors4(r, c)):
            extra_touches += 1
    end_r, end_c = route[-1]
    return (support_run, support_hits, extra_touches, abs(end_r - route[0][0]) + abs(end_c - route[0][1]), len(route))


def _best_route(
    g: np.ndarray,
    seg_start: tuple[int, int],
    seg_col: int,
    L: int,
    fwd: tuple[int, int],
    turn_dir: str,
    obstacle_set: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    best = None
    best_route = None
    for p1 in range(1, L - 1):
        for p2 in range(0, L - p1 - 1):
            p3 = L - 1 - p1 - p2
            if p3 < 1:
                continue
            route, support = _build_candidate_route(seg_start, fwd, turn_dir, L, p1, p2, p3)
            score = _route_score(g, route, support, seg_col, obstacle_set, reserved)
            if score is None:
                continue
            key = (score[0], score[1], score[2], score[3], p2, p1, -p3)
            if best is None or key > best:
                best = key
                best_route = route
    assert best_route is not None, "no route consistent with legend direction"
    return best_route


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    turn_dir, legend_color, legend_cells = _parse_corner_legend(g, bg)
    reserved = set(legend_cells)

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg, legend_color})
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
        neigh = []
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col and v != legend_color:
                    neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a non-bg/non-bar neighbor"

        best_seg = None
        for cand_col, cand_start in neigh:
            for comp in _components(g, cand_col, conn8=True):
                if cand_start in comp:
                    if len(comp) < 2:
                        break
                    deg = _segment_degrees(comp)
                    if deg.get(cand_start, 0) != 1:
                        break
                    path = _segment_main_path(comp, cand_start)
                    L = len(path)
                    key = (L, len(comp), cand_col, cand_start[0], cand_start[1])
                    if best_seg is None or key > best_seg[0]:
                        best_seg = (key, cand_col, cand_start, comp)
                    break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (L, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

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
        assert obstacle_set is not None, "must find an obstacle component"

        fwd = _bar_fwd(bar_comp, h, w)
        new_pts = _best_route(g, seg_start, seg_col, L, fwd, turn_dir, obstacle_set, reserved)

        assert not any(p in obstacle_set for p in new_pts), "output segment overlaps obstacle"
        classes = _step_classes(new_pts, fwd)
        pattern = "".join(classes)
        if turn_dir == "CCW":
            assert set(classes) <= {"FL", "F", "FR"}, "invalid step classes"
            assert "FR" in classes and classes[0] == "FL", f"legend requires FL first, got {pattern}"
            stage = 0
            for cls in classes:
                nxt = {"FL": 0, "F": 1, "FR": 2}[cls]
                assert nxt >= stage, f"non-monotone wrap under CCW legend: {classes}"
                stage = nxt
        else:
            assert set(classes) <= {"FL", "F", "FR"}, "invalid step classes"
            assert "FL" in classes and classes[0] == "FR", f"legend requires FR first, got {pattern}"
            stage = 0
            for cls in classes:
                nxt = {"FR": 0, "F": 1, "FL": 2}[cls]
                assert nxt >= stage, f"non-monotone wrap under CW legend: {classes}"
                stage = nxt

        moves.append((seg_comp, seg_col, new_pts))

    for seg_comp, _seg_col, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 400
PLACEMENT_PAD = 1
LEGEND_PAD = 3

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _legend_cells(turn_dir: str) -> list[tuple[int, int]]:
    if turn_dir == "CCW":
        return [(0, 0), (0, 1), (1, 0)]
    return [(0, 0), (0, 1), (1, 1)]


def _local_first_side(turn_dir: str) -> int:
    # In generator local coords, +side means "left of forward", not "right of forward".
    return 1 if turn_dir == "CCW" else -1


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


def _build_input_path(seg_len: int, turns: int, root_side: int) -> list[tuple[int, int]]:
    remaining = seg_len - 1
    coords = [(0, root_side)]
    diag_step = (-1, root_side)
    orth_step = (-1, 0)
    last_was_diag = None
    turns_left = turns
    while remaining > 0:
        if turns_left <= 0:
            step = diag_step if last_was_diag else orth_step
            if last_was_diag is None:
                step = random.choice([diag_step, orth_step])
            run = remaining
        else:
            if last_was_diag is None:
                is_diag = random.random() < DIAG_START_PROB
            elif last_was_diag:
                is_diag = random.random() < DIAG_CONTINUE_PROB
            else:
                is_diag = random.random() < DIAG_START_PROB
            step = diag_step if is_diag else orth_step
            if last_was_diag is not None and is_diag != last_was_diag:
                turns_left -= 1
            max_run = remaining - turns_left
            run = random.randint(1, max(1, min(max_run, remaining)))
        last_was_diag = step == diag_step
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
        remaining -= run
    return coords


def _build_output_path(
    seg_len: int, root_side: int, turn_dir: str, p1: int, p2: int, p3: int
) -> list[tuple[int, int]]:
    first_side = _local_first_side(turn_dir)
    coords = [(0, root_side)]
    phase_steps = [(1, first_side), (1, 0), (1, -first_side)]
    for phase_len, (dr, dc) in zip([p1, p2, p3], phase_steps):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + dr, pc + dc))
    assert len(coords) == seg_len
    return coords


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _grow_obstacle(
    seeds: list[tuple[int, int]], target_size: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    obstacle = set(seeds)
    while len(obstacle) < target_size:
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


def _make_triple(n: int, turn_dir: str, force_root_side: int | None = None) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    root_side = force_root_side if force_root_side in (-1, 1) else random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    remaining = seg_len - 1
    p1 = random.randint(1, min(PHASE_MAX, remaining - 1))
    remaining -= p1
    p3 = random.randint(1, min(PHASE_MAX, remaining))
    remaining -= p3
    p2 = remaining
    if p2 < 0:
        return None

    output_seg = _build_output_path(seg_len, root_side, turn_dir, p1, p2, p3)
    seg_set = set(output_seg)
    first_side = _local_first_side(turn_dir)
    support = [(output_seg[i][0], output_seg[i][1] - first_side) for i in range(p1, p1 + p2 + 1)]
    if any(p in seg_set or p in bar_set for p in support):
        return None

    min_obs = max(OBS_SIZE_MIN, len(support))
    obs_size = random.randint(min_obs, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(support, obs_size, seg_set | bar_set)

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (2 <= nearest_md <= 6):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, root_side)
    if _is_4adj(input_seg, obstacle):
        return None

    union = bar + obstacle + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
        "root_side": root_side,
        "support_len": len(support),
    }


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _place_triple(
    triple: dict, edge: str, n: int, reserved: np.ndarray
) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_seg"]
    bar_len = triple["bar_len"]

    for pos in positions:
        union_g = _transform(union, edge, n, bar_len, pos)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        placed = {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "root_side": triple["root_side"],
            "reserve": (r0, c0, r1, c1),
            "support_len": triple["support_len"],
        }
        return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 11:
        max_triples = 1
    elif n < 17:
        max_triples = 2
    else:
        max_triples = 3
    triple_count = random.randint(1, max_triples)

    palette = random.sample(range(10), 5)
    bg, bar_color, seg_color, obstacle_color, legend_color = palette
    turn_dir = random.choice(["CW", "CCW"])

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    reserved[:LEGEND_PAD, :LEGEND_PAD] = True
    placed_triples = []

    paint_coords(inp, _legend_cells(turn_dir), legend_color)
    paint_coords(out, _legend_cells(turn_dir), legend_color)

    first_side = _local_first_side(turn_dir)
    for idx, edge in enumerate(random.sample(list(EDGE_DATA), triple_count)):
        force_root_side = -first_side if idx == 0 else None
        placed = None
        for _ in range(40):
            triple = _make_triple(n, turn_dir, force_root_side=force_root_side)
            if triple is None:
                continue
            placed = _place_triple(triple, edge, n, reserved)
            if placed is not None:
                break
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
